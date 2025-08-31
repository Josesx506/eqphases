import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import seisbench.models as sbm


def load_phasenet():
    """
    Load the original pre-trained phasenet model from seisbench
    """
    phnt = sbm.PhaseNet.from_pretrained("original")
    phnt.eval()
    return phnt


def load_eqtransformer():
    """
    Load the original pre-trained eqtransformer model from seisbench
    """
    eqtr = sbm.EQTransformer.from_pretrained("stead")
    eqtr.eval()
    return eqtr