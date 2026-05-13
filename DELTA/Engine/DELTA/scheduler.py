import torch
import time
from typing import List, Dict, Optional, Tuple
import numpy as np
import random


class PageManager:
    """Simple randomized page allocator.
    Keeps a global pool of free page indices and hands out random pages.
    """
    def __init__(self, total_pages: int, seed: Optional[int] = None):
        self.total_pages = int(total_pages)
        self.free: List[int] = list(range(self.total_pages))
        self.in_use = set()
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.free)

    def alloc_one(self) -> int:
        if not self.free:
            raise RuntimeError("Out of KV pages: pool is empty")
        p = self.free.pop() # already shuffled for randomness
        self.in_use.add(p)
        return p

    def alloc(self, n: int) -> List[int]:
        if n > len(self.free):
            raise RuntimeError(f"Out of KV pages: need {n}, have {len(self.free)} left")
        # sample without replacement from shuffled pool by popping
        return [self.alloc_one() for _ in range(n)]

    def free_pages(self, pages: List[int]):
        # return to pool and lightly reshuffle tail for randomness
        for p in pages:
            if p in self.in_use:
                self.in_use.remove(p)
                self.free.append(p)
        # keep it simple; occasional shuffle keeps distribution random enough
        random.shuffle(self.free)


class Request:
    def __init__(self, id: int, input_ids: torch.LongTensor, max_length: int, eot_token_ids: list, page_manager: PageManager) -> None:
        self.id = id
        self.input_ids = input_ids
        self.max_length = max_length
        self.eot_token_ids = eot_token_ids
        self.output_ids = []
        self.page_manager = page_manager
        
        self.status = "waiting"
        
        self.page_indices = []
        self.page_lastlen = 0

        self.start_time = None
        self.end_time = None

    def plan(self, qo_len: int, page_size: int):
        """Add tokens and allocate pages as needed"""
        
        self.page_lastlen += qo_len
        while self.page_lastlen >= page_size:
            self.page_indices.append(self.page_manager.alloc_one())
            self.page_lastlen -= page_size

    def initialize_pages(self, slot_id: int, pages_per_slot: int):
        """Initialize page allocation for this request"""
        self.status = "active"
        self.start_time = time.perf_counter()
        
        self.page_indices = [self.page_manager.alloc_one()]
        self.page_lastlen = 0
        
    def get_qo_tokens(self):
        if len(self.output_ids) == 0:
            return self.input_ids
        else:
            return torch.tensor([self.output_ids[-1]], dtype=torch.long)

    def append_output(self, token_id: int):
        self.output_ids.append(token_id)
        
        if token_id in self.eot_token_ids or (len(self.input_ids) + len(self.output_ids)) >= self.max_length:
            self.page_manager.free_pages(self.page_indices)
            self.status = "finished"
            self.end_time = time.perf_counter()
            return True
        return False

    def is_finished(self):
        return self.status == "finished"


class BatchScheduler:
    def __init__(self, batch_size: int, device: str, is_main_process: bool, pages_per_slot: int, page_size: int, page_manager: PageManager):
        self.is_main_process = is_main_process
        self.pages_per_slot = pages_per_slot
        self.page_size = page_size
        self.page_manager = page_manager
        self.max_pages = 0
        self.active_bsz = 0
        
        self.request_queue = []
        self.waiting_queue_ids = []
        self.runner_slots = torch.tensor([-1] * batch_size)
        
        self.stats = {
            'total_tokens_generated': 0,
            'start_time': None,
            'encode_steps': 0,
            'decode_steps': 0
        }

    def add_requests(self, input_ids_list: list, max_lengths: list, eot_token_ids: list):
        for i, input_ids in enumerate(input_ids_list):
            max_length = max_lengths[i] if isinstance(max_lengths, list) else max_lengths
            new_request_id = len(self.request_queue)
            request = Request(new_request_id, input_ids, max_length, eot_token_ids, self.page_manager)
            self.waiting_queue_ids.append(new_request_id)
            self.request_queue.append(request)
        if self.is_main_process:
            print(f"Added {len(input_ids_list)} requests. Total waiting: {len(self.waiting_queue_ids)}")

    def schedule_requests(self) -> Tuple[List[int], List[Request]]:
        newly_scheduled_slot_ids = []
        newly_scheduled_requests = []
        
        for slot_id in range(len(self.runner_slots)):
            if self.runner_slots[slot_id] == -1 and self.waiting_queue_ids:
                request_id = self.waiting_queue_ids.pop(0)
                request = self.request_queue[request_id]
                
                self.runner_slots[slot_id] = request_id
                newly_scheduled_slot_ids.append(slot_id)
                newly_scheduled_requests.append(request)
        
        return newly_scheduled_slot_ids, newly_scheduled_requests

    def check_finished_requests(self) -> List[int]:
        finished_slots = []
        
        for slot_id in range(len(self.runner_slots)):
            if self.runner_slots[slot_id] != -1:
                request_id = self.runner_slots[slot_id]
                request = self.request_queue[request_id]
                
                if request.is_finished():
                    finished_slots.append(slot_id)
                    self.runner_slots[slot_id] = -1
        
        return finished_slots

    def has_active_requests(self) -> bool:
        return any(x != -1 for x in self.runner_slots)

    def get_detailed_results(self, tokenizer=None):
        results = []
        for request in self.request_queue:
            if not request.is_finished():
                continue
            
            total_time = request.end_time - request.start_time if request.end_time and request.start_time else 0
            
            result = {
                'request_id': request.id,
                'status': request.status,
                'input_length': len(request.input_ids),
                'output_length': len(request.output_ids),
                'tokens_generated': len(request.output_ids),
                'total_time': total_time,
                'tokens_per_second': len(request.output_ids) / total_time if total_time > 0 else 0,
            }
            
            if tokenizer:
                try:
                    result['input_text'] = tokenizer.decode(request.input_ids, skip_special_tokens=True)
                    result['output_text'] = tokenizer.decode(request.output_ids, skip_special_tokens=True)
                except:
                    result['input_text'] = f"[{len(request.input_ids)} tokens]"
                    result['output_text'] = f"[{len(request.output_ids)} tokens]"
            
            results.append(result)
        
        return sorted(results, key=lambda x: x['request_id'])
