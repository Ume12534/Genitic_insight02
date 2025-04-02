# Genitic_insight/forms.py
from django import forms
from .models import FastaFile


class FastaUploadForm(forms.Form):
    file = forms.FileField()
    sequenceType = forms.CharField()
    descriptor = forms.CharField()

class FastaUploadForm(forms.ModelForm):
    class Meta:
        model = FastaFile
        fields = ['file']
