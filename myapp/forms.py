from django import forms  
class UploadForm(forms.Form):   
    file = forms.FileField() 

class SendEmailForm(forms.Form):
    name = forms.CharField(required=True)
    subject = forms.CharField(required=True)
    email = forms.EmailField(required=True)
    message = forms.CharField(widget=forms.Textarea, required=True)