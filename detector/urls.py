from django.urls import path
from .views import RegisterView, LoginView, LogoutView, PredictView
from django.contrib.auth.decorators import login_required
from .views import PredictView

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path("predict/", login_required(PredictView.as_view()), name="predict"),
]
