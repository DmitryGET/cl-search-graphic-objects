from django import forms
from .models import Photo


class PhotoUploadForm(forms.ModelForm):
    class Meta:
        model = Photo
        fields = ["image"]

    CHOICES = [
        ('YOLO', 'YOLO'),
        ('FastRCNN', 'FastRCNN')
    ]

    model = forms.ChoiceField(choices=CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
    image = forms.ImageField(label="Выберите фото")
