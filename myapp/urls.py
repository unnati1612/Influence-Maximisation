from django.contrib import admin
from django.urls import path
from myapp import views
urlpatterns=[
    path("",views.index,name='home'),    
    path("submitform",views.showdata,name='submitform'),
    path("centrality",views.centrality,name='centrality'),
    path("linearThr",views.linearThr,name='linearThr'),
    path("icThr",views.icThr,name='icThr'),
    path("aboutProj",views.aboutProj,name='aboutProj'),
    ]