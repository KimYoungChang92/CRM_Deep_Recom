
# from django.contrib import admin
# from django.urls import path
#
# urlpatterns = [
#     path('admin/', admin.site.urls),
# ]

from django.urls import path

from django.conf.urls import url, include
from django.contrib import admin
from rest_framework import routers
from rest_framework_swagger.views import get_swagger_view

import member.api
import member.views as views

app_name='member'

router = routers.DefaultRouter()
router.register('members', member.api.MemberViewSet)

urlpatterns = [

    #url(r'^admin/', admin.site.urls),
    url(r'^api/doc', get_swagger_view(title='Rest API Document')),
    url(r'^api/v1/', include((router.urls, 'member'), namespace='api')),

    path('admin/',admin.site.urls),
    # path('predict/', views.predict, name='predict'),


    #add test1
    #path('torch_predict/',views.torch_predict, name='torch_predict'),
    #add test2
    #path('recommend/user_id/<int:user_id>.<int:top_n>',views.predict_rating, name='recommend/user_id/'),


    path('recommend/product/<int:REAL_USER_ID>.<int:top_n>',views.deepfm, name='recommend/product/')

]