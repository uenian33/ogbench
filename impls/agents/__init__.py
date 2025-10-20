from agents.crl import CRLAgent
from agents.crl_infonce import CRLInfoNCEAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.rws import RWSAgent
from agents.sac import SACAgent
from agents.td_infonce import TDInfoNCEAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    rws=RWSAgent,
    sac=SACAgent,
    td_infonce=TDInfoNCEAgent,
    crl_infonce=CRLInfoNCEAgent,
)
