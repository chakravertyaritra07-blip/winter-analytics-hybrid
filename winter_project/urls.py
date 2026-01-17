from django.contrib import admin
from django.urls import path
from core.views import dashboard  # Import your view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', dashboard, name='home'),  # Map the empty path '' to your dashboard
]