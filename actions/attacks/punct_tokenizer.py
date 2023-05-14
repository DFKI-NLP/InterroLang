from OpenAttack.text_process.tokenizer.base import Tokenizer
from OpenAttack.data_manager import DataManager
from OpenAttack.tags import *
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag.perceptron import PerceptronTagger

_POS_MAPPING = {
    "JJ": "adj",
    "VB": "verb",
    "NN": "noun",
    "RB": "adv"
}


class PunctTokenizer(Tokenizer):
    """
    Tokenizer based on nltk.word_tokenizer.

    :Language: english
    """


    TAGS = { TAG_English }

    def __init__(self) -> None:
        self.sent_tokenizer = sent_tokenize #DataManager.load("TProcess.NLTKSentTokenizer")
        self.word_tokenizer = word_tokenize #nltk.WordPunctTokenizer().tokenize
        self.pos_tagger = PerceptronTagger().tag #DataManager.load("TProcess.NLTKPerceptronPosTagger")
        
    def do_tokenize(self, x, pos_tagging=True):
        sentences = self.sent_tokenizer(x)
        tokens = []
        for sent in sentences:
            tokens.extend( self.word_tokenizer(sent) )

        if not pos_tagging:
            return tokens
        ret = []
        for word, pos in self.pos_tagger(tokens):
            if pos[:2] in _POS_MAPPING:
                mapped_pos = _POS_MAPPING[pos[:2]]
            else:
                mapped_pos = "other"
            ret.append( (word, mapped_pos) )
        return ret

    def do_detokenize(self, x):
        return " ".join(x)
