import matplotlib.pyplot as plt
from data.bcs_envelope import BcsEnvelope

envelope=BcsEnvelope()
envelope.size_env = (4, 4)
envelope.grafico_envelope2()
plt.show()
