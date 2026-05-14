import os

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.views import View
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.core.files.storage import default_storage

from .utils import predict_image


# --------------------
# REGISTER VIEW
# --------------------
class RegisterView(View):
    def get(self, request):
        return render(request, "register.html")

    def post(self, request):
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")

        if password != confirm_password:
            return render(request, "register.html", {"error": "Passwords do not match"})

        if User.objects.filter(username=username).exists():
            return render(request, "register.html", {"error": "Username already exists"})

        User.objects.create_user(username=username, email=email, password=password)
        return redirect("login")


# --------------------
# LOGIN VIEW
# --------------------
class LoginView(View):
    def get(self, request):
        return render(request, "login.html")

    def post(self, request):
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect("predict")

        return render(request, "login.html", {"error": "Invalid username or password"})


# --------------------
# LOGOUT VIEW
# --------------------
class LogoutView(View):
    def get(self, request):
        logout(request)
        return redirect("login")


# --------------------
# PREDICT VIEW
# --------------------
@method_decorator(login_required, name='dispatch')
class PredictView(View):

    def get(self, request):
        return render(request, "index.html")

    def post(self, request):
        context = {}

        if "image" not in request.FILES:
            context["error"] = "No image uploaded."
            return render(request, "index.html", context)

        uploaded_image = request.FILES["image"]

        ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp']
        MAX_SIZE_MB = 10

        if uploaded_image.content_type not in ALLOWED_TYPES:
            context["error"] = "Only JPEG, PNG, and WebP images are supported."
            return render(request, "index.html", context)

        if uploaded_image.size > MAX_SIZE_MB * 1024 * 1024:
            context["error"] = f"Image must be under {MAX_SIZE_MB}MB."
            return render(request, "index.html", context)

        # Save via default_storage (Cloudinary in production)
        filename = default_storage.save(uploaded_image.name, uploaded_image)
        img_url = default_storage.url(filename)

        try:
            prediction_result = predict_image(img_url)

            context.update({
                "img_url": img_url,
                "result": prediction_result["result"],
                "confidence": prediction_result["confidence"],
                "fake_percent": prediction_result["details"]["Fake"],
                "real_percent": prediction_result["details"]["Real"],
                "success": True
            })

        except Exception as e:
            context["error"] = f"Prediction failed: {str(e)}"

        finally:
            # Clean up from Cloudinary after prediction
            try:
                default_storage.delete(filename)
            except Exception:
                pass

        return render(request, "index.html", context)