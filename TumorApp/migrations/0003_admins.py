# Generated by Django 2.1.7 on 2024-07-24 13:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('TumorApp', '0002_auto_20240723_1534'),
    ]

    operations = [
        migrations.CreateModel(
            name='Admins',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=40)),
                ('password', models.CharField(max_length=25)),
            ],
        ),
    ]
