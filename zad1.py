import math
from datetime import datetime

def calculate_days_lived(birth_date):
    today = datetime.today()
    delta = today - birth_date
    return delta.days

def biorhythm_value(cycle_length, days_lived):
    return math.sin(2 * math.pi * days_lived / cycle_length)

def check_biorhythm_status(biorhythm, next_biorhythm, name, type):
    if biorhythm > 0.5:
        print(f"Twoj {type} biorytm jest wysoki ({biorhythm:.2f}). Gratulacje, {name}!")
    elif biorhythm < -0.5:
        print(f"Twoj {type} biorytm jest niski ({biorhythm:.2f}). To nie jest najlepszy dzien, {name}.")
        if next_biorhythm > biorhythm:
            print("Nie martw sie. Jutro będzie lepiej!")
        else:
            print("Niestety, jutro moze byc podobnie, ale trzymaj sie!")

name = input("Podaj swoje imie: ")
year = int(input("Podaj rok urodzenia: "))
month = int(input("Podaj miesiąc urodzenia (1-12): "))
day = int(input("Podaj dzień urodzenia (1-31): "))

birth_date = datetime(year, month, day)
days_lived = calculate_days_lived(birth_date)

physical_today = biorhythm_value(23, days_lived)
emotional_today = biorhythm_value(28, days_lived)
intellectual_today = biorhythm_value(33, days_lived)

physical_tomorrow = biorhythm_value(23, days_lived + 1)
emotional_tomorrow = biorhythm_value(28, days_lived + 1)
intellectual_tomorrow = biorhythm_value(33, days_lived + 1)

print(f"\nWitaj, {name}!")
print(f"Dzis jest {days_lived} dzien twojego życia.")
print(f"Twoje wyniki biorytmów na dzis:")
print(f"Fizyczny: {physical_today:.2f}")
print(f"Emocjonalny: {emotional_today:.2f}")
print(f"Intelektualny: {intellectual_today:.2f}")

# Sprawdzanie biorytmów
check_biorhythm_status(physical_today, physical_tomorrow, name, "fizyczny")
check_biorhythm_status(emotional_today, emotional_tomorrow, name, "emocjonalny")
check_biorhythm_status(intellectual_today, intellectual_tomorrow, name, "intelektualny")

