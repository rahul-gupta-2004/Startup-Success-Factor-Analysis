import streamlit as st
from streamlit_option_menu import option_menu
import dashboard, model

st.set_page_config(
    page_title = "Startup Analysis",
    layout = "centered",
)

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title, 
            "function": func
        })

    def run():
        with st.sidebar:        
            app = option_menu(
                menu_title='Main Menu',
                options=['Dashboard', 'Model'],
                default_index=0,
                menu_icon="list",
                icons=['graph-up', 'cpu']
            )
        
        if app == "Dashboard":
            dashboard.app()
        if app == "Model":
            model.app()
             
    run()