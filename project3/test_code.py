import numpy as np
class Dog:
    """A simple attempt to model a dog."""

    def __init__(self, name, age):
        """Initialize name and age attributes."""
        self.name = name
        self.age = age

    def sit(self):
        """Simulate a dog sitting in response to a command."""
        print(f"{self.name} is now sitting.")

    def roll_over(self):

        """Simulate rolling over in response to a command."""
        print(f"{self.name} rolled over!")

my_dog = Dog(name='Jerry', age=6)
print(f"My dog's name is {my_dog.name}")
print(f"My dog is {my_dog.age} years old")

my_dog.sit()
my_dog.roll_over()

class Car:
    """Creating a class"""
    def __init__(self, make, model, y):
        self.make = make
        self.model = model
        self.year = y

    def get_descriptive_name(self):
        long_name = f"{self.year} {self.make} {self.model}"
        return long_name.title()

my_car = Car("Mercedes","gl", y=2021)
print(my_car.get_descriptive_name())

x = np.random.randn(20)
print(x)
x1 = x[0:10]
x2 = x[10:]
print(x1)
print(x2)

iv = np.array([[1], [2]])
print(iv)
print((np.tile(np.transpose(iv), (3, 1))))
