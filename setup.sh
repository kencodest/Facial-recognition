mkdir -p ~/.streamlit/

echo "\
[theme]
primaryColor = '#d33682'
backgroundColor = '#002b36'
secondaryBackgroundColor = '#586e75'
textColor = '#fafafa'
font = 'sans serif'
[server]
port = $PORT
enableCORS = false
headless = true
" > ~/.streamlit/config.toml
