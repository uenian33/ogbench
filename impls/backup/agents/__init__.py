from agents.crl import CRLAgent
from agents.crl_infonce import CRLInfoNCEAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.rws import RWSAgent
from agents.td_rws import TDRWSAgent
from agents.sac import SACAgent
from agents.td_infonce import TDInfoNCEAgent
from agents.expect_rws import ExpectileStepsAgent
agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    rws=RWSAgent,
    td_rws=TDRWSAgent,
    sac=SACAgent,
    td_infonce=TDInfoNCEAgent,
    crl_infonce=CRLInfoNCEAgent,
    expectile_steps=ExpectileStepsAgent
)
