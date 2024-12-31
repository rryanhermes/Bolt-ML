from django.shortcuts import redirect
from django.contrib import messages
from functools import wraps

def premium_required(view_func):
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            messages.warning(request, "Please log in to access this feature.")
            return redirect('login')
        
        if not request.user.userprofile.is_premium:
            messages.warning(request, "This feature requires a premium subscription.")
            return redirect('premium')
            
        return view_func(request, *args, **kwargs)
    return _wrapped_view 