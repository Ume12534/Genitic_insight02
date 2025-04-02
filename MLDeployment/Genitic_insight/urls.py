from django.urls import path
from .import views

urlpatterns = [
    path('', views.home, name='home'),
    
    path('feature_extraction/', views.feature_extraction, name='feature_extraction'),

     path('feature_extraction/analyze/', views.analyze_sequence, name='analyze_sequence'),
    
    path('data-visualization/', views.data_visualization, name='data_visualization'),
    
    path('module-selection/', views.module_selection, name='module_selection'),
    
    path('evaluation-values/', views.evaluation_values, name='evaluation_values'),

    # path('save_extracted_data/', views.save_extracted_data, name='save_extracted_data'),

    path('train-model/', views.train_model, name='train_model'),

    # path('training-results/', views.training_results, name='training_results'),

    # path('predict/', views.make_predictions, name='make_predictions'),
]
