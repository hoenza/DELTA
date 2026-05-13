import json
from pathlib import Path


def _token_content(value):
    if isinstance(value, dict):
        return value.get("content")
    return value


def _iter_tokenizer_dirs(model_path, model_name):
    seen = set()

    def normalize(path_like):
        path = Path(path_like).expanduser()
        if path.is_file():
            path = path.parent
        return path

    for candidate in (model_path, model_name):
        if candidate is None:
            continue
        path = normalize(candidate)
        if not path.is_dir():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        yield path


def _load_raw_fast_tokenizer(tokenizer_dir: Path):
    from transformers import PreTrainedTokenizerFast

    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        return None

    tokenizer_config = {}
    tokenizer_config_path = tokenizer_dir / "tokenizer_config.json"
    if tokenizer_config_path.is_file():
        with tokenizer_config_path.open() as f:
            tokenizer_config = json.load(f)

    kwargs = {}
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        value = _token_content(tokenizer_config.get(key))
        if value is not None:
            kwargs[key] = value

    model_max_length = tokenizer_config.get("model_max_length")
    if isinstance(model_max_length, int):
        kwargs["model_max_length"] = model_max_length

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_json), **kwargs)

    for attr in ("clean_up_tokenization_spaces", "padding_side", "truncation_side"):
        if attr in tokenizer_config:
            setattr(tokenizer, attr, tokenizer_config[attr])

    # Preserve tokenizer-time BOS/EOS behavior from tokenizer_config.json.
    # The historical DEL3 path used AutoTokenizer under transformers 4.47,
    # which honored add_bos_token for the Qwen tokenizer. Rebuilding directly
    # from tokenizer.json drops that flag unless we copy it explicitly.
    for attr in ("add_bos_token", "add_eos_token"):
        if attr in tokenizer_config:
            setattr(tokenizer, attr, tokenizer_config[attr])

    chat_template = tokenizer_config.get("chat_template")
    if chat_template is not None:
        tokenizer.chat_template = chat_template

    tokenizer._resolved_source = str(tokenizer_dir)
    tokenizer._resolved_loader = "tokenizer.json"
    return tokenizer


def load_tokenizer(model_path, model_name, verbose=True):
    from transformers import AutoTokenizer

    last_error = None

    for tokenizer_dir in _iter_tokenizer_dirs(model_path, model_name):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_dir),
                trust_remote_code=True,
                local_files_only=True,
            )
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer._resolved_source = str(tokenizer_dir)
            tokenizer._resolved_loader = "AutoTokenizer(local)"
            if verbose:
                print(f"Loaded tokenizer from local directory {tokenizer_dir}")
            return tokenizer
        except Exception as exc:
            last_error = exc

        try:
            # Fallback only when AutoTokenizer cannot be reconstructed locally.
            # Rebuilding directly from tokenizer.json can drop model-specific
            # post-processing such as implicit BOS insertion.
            tokenizer = _load_raw_fast_tokenizer(tokenizer_dir)
            if tokenizer is not None:
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                if verbose:
                    print(f"Loaded tokenizer from {tokenizer_dir / 'tokenizer.json'}")
                return tokenizer
        except Exception as exc:
            last_error = exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_name),
            trust_remote_code=True,
        )
    except Exception as exc:
        if last_error is not None:
            raise RuntimeError(
                f"Failed to load tokenizer from local checkpoint assets and model_name={model_name!r}"
            ) from last_error
        raise exc

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer._resolved_source = str(model_name)
    tokenizer._resolved_loader = "AutoTokenizer(model_name)"
    if verbose:
        print(f"Loaded tokenizer from model_name={model_name}")
    return tokenizer
