from django.shortcuts import render, redirect
from .forms import PhotoUploadForm
from .for_api import single_image_prediction


def upload_photo(request):
    if request.method == "POST":
        form = PhotoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            photo = form.save()

            input_image_path = photo.image.path
            single_image_prediction(
                input_image_path, r"media\processed_photos\processed_image.jpg", 0.5
            )

            photo.save()
            return redirect(
                "view_processed_photo", r"\media\processed_photos\processed_image.jpg"
            )
    else:
        form = PhotoUploadForm()

    return render(request, "upload_photo.html", {"form": form})


def view_processed_photo(request, photo_path):
    print(photo_path)
    return render(request, "view_processed_photo.html", {"photo": photo_path})
