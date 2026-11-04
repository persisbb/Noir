"""
URL configuration for noir_site project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""


from django.contrib import admin
from django.urls import path
from chat.views import (
    home,
    ask,
    sources,
    source_explorer,
    source_detail,
    evaluations,
    documents,
    ask_document
)

from django.contrib.auth.decorators import login_required
from django.urls import include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("django.contrib.auth.urls")),
    path("", home, name="home"),
    path("ask/", login_required(ask), name="ask"),
    path("sources/", login_required(sources), name="sources"),
    path("sources/explore/", login_required(source_explorer), name="source_explorer"),
    path("sources/chunk/<str:vector_id>/", login_required(source_detail), name="source_detail"),
    path("evaluations/", login_required(evaluations), name="evaluations"),
    path("documents/", login_required(documents), name="documents"),
    path("documents/ask/", login_required(ask_document), name="ask_document"),
]

