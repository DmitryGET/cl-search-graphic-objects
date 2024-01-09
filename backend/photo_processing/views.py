from django.conf import settings
from django.shortcuts import render, redirect
from .forms import PhotoUploadForm
from .for_api import single_image_prediction
from os import walk


def upload_photo(request):
    if request.method == "POST":
        form = PhotoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            model = form.cleaned_data["model"]
            photo = form.save()

            input_image_path = photo.image.path
            res = single_image_prediction(
                input_image_path,
                r"media\processed_photos\processed_image.jpg",
                0.75,
                model,
            )

            photo.save()
            if res == "Fast":
                return redirect(
                    "view_processed_photo",
                    r"\media\processed_photos\processed_image.jpg",
                )
            else:
                filename = next(
                    walk(settings.MEDIA_ROOT + r"\processed_photos\pic"),
                    (None, None, []),
                )[2][0]
                return redirect(
                    "view_processed_photo",
                    r"\media\processed_photos\pic" + "\\" + filename,
                )
    else:
        form = PhotoUploadForm()

    return render(request, "upload_photo.html", {"form": form})


def view_processed_photo(request, photo_path):
    print(photo_path)
    return render(request, "view_processed_photo.html", {"photo": photo_path})
