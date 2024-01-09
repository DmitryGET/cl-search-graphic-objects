from django.urls import path
from . import views


urlpatterns = [
    path("upload_photo/", views.upload_photo, name="upload_photo"),
    path(
        "view_processed_photo/<str:photo_path>/",
        views.view_processed_photo,
        name="view_processed_photo",
    ),
]
