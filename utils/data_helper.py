import re
from pathlib import Path
import yaml
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator






_patterns = [r"\'", r"\"", r"\.", r"<br \/>", r",", r"\(", r"\)", r"\!", r"\?", r"\;", r"\:", r"\s+"]

_replacements = [" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "]

_patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))

def _basic_arabic_normalize(line):
    r"""
    Basic normalization for a line of text.
    Normalization includes
    - lowercasing
    - complete some basic text normalization for English words as follows:
        add spaces before and after '\''
        remove '\"',
        add spaces before and after '.'
        replace '<br \/>'with single space
        add spaces before and after ','
        add spaces before and after '('
        add spaces before and after ')'
        add spaces before and after '!'
        add spaces before and after '?'
        replace ';' with single space
        replace ':' with single space
        replace multiple spaces with single space

    Returns a list of tokens after splitting on whitespace.
    """
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()


###################################################################################################################################################################################################

def _get_tokenizers():
    
    english_tokenizer = get_tokenizer("basic_english")
    arabic_tokenizer  = _basic_arabic_normalize
    
    return english_tokenizer,arabic_tokenizer

def _build_vocab(data_itr,tokenizer):
    
    config_yaml_path = ( Path(__file__) / "../../config.yaml" ).resolve()
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    
    v = build_vocab_from_iterator(map(tokenizer,data_itr),min_freq=config["min_freq"],specials=["<unk>","<SOS>","<EOS>","<PAD>"])
    v.set_default_index(v["<unk>"])
    
    return v


