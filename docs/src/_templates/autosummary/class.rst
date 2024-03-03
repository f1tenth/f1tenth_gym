{% extends "!autosummary/class.rst" %}

{% block methods %} {% if methods %}

   .. autosummary::
    :toctree: generated/
        {% for item in methods %}
            {% if item != '__init__'%}
              ~{{ name }}.{{ item }}
            {% endif %}
        {%- endfor %}

{% endif %} {% endblock %}

{% block attributes %} {% if attributes %}

{% endif %} {% endblock %}