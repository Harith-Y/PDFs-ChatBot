import streamlit as st

def get_pdf_text(text):
    return text

def main():
    st.set_page_config(
        page_title="PDF-ChatBot",
        page_icon=":books:"
    )

    st.header("Chat with PDFs :books:")
    st.text_input("Type your Query about your documents: ")

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'")
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                st.write(raw_text)
            st.success("Done!")

if __name__ == '__main__':
    main()