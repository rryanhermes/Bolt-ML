from django.conf import settings

def analytics(request):
    """
    Add Google Analytics tracking ID to the context
    """
    return {
        'ga_tracking_id': settings.GA_TRACKING_ID
    } 