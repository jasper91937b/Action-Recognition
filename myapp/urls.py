from django.urls import path

from . import views

app_name = 'action'

urlpatterns = [
    path('predict/',views.predict),
    path("send_data/", views.send_data)
]