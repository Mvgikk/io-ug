import math

def forwardPass(wiek, waga, wzrost):
    hidden1 = wiek * -0.46122 + waga * 0.97314 + wzrost * -0.39203 + 0.80109
    hidden1_po_aktywacji = act(hidden1)
    hidden2 =  wiek * 0.78548 + waga * 2.10584 + wzrost * -0.57847
    hidden2_po_aktywacji = act(hidden2)
    output = (hidden1_po_aktywacji * -0.81546 + hidden2_po_aktywacji * 1.03775) -0.2368
    return output

def act(x):
    return 1/(1+(math.e)**(-x))

#  = 0.798528
print(f"Wynik  = {forwardPass(23,75,176)}")

#  = -0.0145181
print(f"Wynik  = {forwardPass(28,120,175)}")