from typing import List
import tensorflow as tf
import sentencepiece as sp

_RESERVED_TOKENS = 103

def create_encoder(vocab_file:str):
    return SentencePieceEncoder(vocab_file)

class SentencePieceEncoder(object):
    
    def __init__(self, model_file : str, 
                 reserved_tokens: int = _RESERVED_TOKENS): 
        self._tokenizer = sp.SentencePieceProcessor()
        self._sp_model = tf.io.gfile.GFile(model_file,"rb").read()
        self._tokenizer.LoadFromSerializedProto(self._sp_model)
        self._reserved_tokens = reserved_tokens
    
    def tokenize(self,text:str) -> List[int]:
        ids = self._tokenizer.EncodeAsIds(text)
        ids = [i + self._reserved_tokens if i >1 else i for i in ids]
        return ids
    
    def detokenize(self, ids:list[int]) -> str:
        ids = [i - self._reserved_tokens if i>103 + 1 else i for i in ids]
        text = self._tokenizer.DecodeIds(ids)
        return text
    
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.GetPieceSize() + self._reserved_tokens
    