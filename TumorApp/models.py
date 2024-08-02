from django.db import models

class Patient(models.Model):
    name = models.CharField(max_length=100,unique=True)
    result = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)
    sex = models.CharField(max_length = 6,default="Male")
    phonenumber = models.CharField(max_length = 14,default="+919767345241")
    age = models.IntegerField(null=True,default=21)

    def __str__(self):
        return f"{self.name}"

class Admins(models.Model):
    username = models.CharField(max_length = 40,unique=True)
    password = models.CharField(max_length=25)
    security_question = models.CharField(max_length=255,default="what is your friend name? ")
    security_answer = models.CharField(max_length=255,default="chakri")

    def __str__(self):
        return f"{self.username}"
