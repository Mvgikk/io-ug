import math
import random
import numpy as np
import matplotlib.pyplot as plt

v0 = 50
h = 100
g = 9.81

def calculate_distance(angle_degrees):
    angle_radians = math.radians(angle_degrees)
    return (v0 * math.cos(angle_radians) / g) * (
        v0 * math.sin(angle_radians) + math.sqrt(v0 ** 2 * math.sin(angle_radians) ** 2 + 2 * g * h))


def calculate_trajectory(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    
    t_flight = (v0 * np.sin(angle_radians) + np.sqrt(v0 ** 2 * np.sin(angle_radians) ** 2 + 2 * g * h)) / g
    
    t = np.linspace(0, t_flight, num=500)
    
    x = v0 * np.cos(angle_radians) * t
    y = h + v0 * np.sin(angle_radians) * t - 0.5 * g * t ** 2
    
    return x, y

def plot_trajectory(x, y):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Trajectoria pocisku", color="blue")

    plt.title("Projectile Motion for the Trebuchet")
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")

    plt.grid(True)

    plt.xlim(0, 350)
    plt.ylim(0, 160)

    plt.savefig('trajektoria.png')

    plt.show()


target_distance = random.randint(50, 340)
print(f"Cel znajduje się w odległości: {target_distance} metrów.")

attempts = 0
while True:
        angle = float(input("Podaj kąt strzału (w stopniach): "))
        attempts += 1
        
        distance = calculate_distance(angle)
        print(f"Pocisk przeleciał {distance:.2f} metrów.")
        
        if target_distance - 5 <= distance <= target_distance + 5:
            print(f"Cel trafiony! Liczba prób: {attempts}")
            
            x, y = calculate_trajectory(angle)
            
            plot_trajectory(x, y)
            break
        else:
            print("Chybiony! Spróbuj ponownie.")
