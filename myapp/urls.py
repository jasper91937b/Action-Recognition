from django.urls import path

from . import views

app_name = 'myapp'

urlpatterns = [
    path('predict/<int:time>',views.predict, name = "predict"),
    path("send_data/", views.send_data, name = "data"),
    path("img/<int:time>", views.img, name = "img"),
    path("stoppredict/<int:time>", views.stop_predict, name = "stop"),
    path("sendemail/", views.SendEmail, name = "email"),
    path('',views.predict, name = "index"),
]