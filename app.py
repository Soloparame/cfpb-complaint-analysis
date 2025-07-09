import streamlit as st
from src.rag.pipeline import generate_answer

st.set_page_config(page_title="CrediTrust Complaint Assistant", page_icon="📊")

st.title("📊 CrediTrust Complaint Assistant")
st.write("Ask a question about financial complaints (Credit Cards, BNPL, etc).")

# Text input
query = st.text_input("Enter your question:", placeholder="e.g. What issues do users report about BNPL?")

# Ask button
if st.button("Ask") and query.strip() != "":
    with st.spinner("Thinking..."):
        answer, sources = generate_answer(query)

        # Display answer
        st.subheader("📬 Answer:")
        st.success(answer)

        # Show sources
        st.subheader("📚 Sources Used:")
        for i, (chunk, meta) in enumerate(sources[:3]):
            st.markdown(f"**{i+1}. {meta['product']}**")
            st.markdown(f"> {chunk[:500]}...")

# Clear input/output
if st.button("Clear"):
    st.experimental_rerun()
