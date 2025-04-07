import streamlit as st
from huggingface_hub import list_datasets
from datasets import load_dataset, get_dataset_config_names
import pandas as pd
import json
import zipfile
from io import BytesIO
from fpdf import FPDF

def export_dataset(dataset_id, output_format="best"):
    output_files = []

    try:
        try:
            dataset = load_dataset(dataset_id)
        except Exception as e:
            if "Config name is missing" in str(e):
                config_names = get_dataset_config_names(dataset_id)
                if config_names:
                    dataset = load_dataset(dataset_id, config_names[0])
                else:
                    raise ValueError("No configs available.")
            else:
                raise e

        for split in dataset.keys():
            try:
                df = dataset[split].to_pandas()

                dataset_path = dataset_id.split("/")
                if len(dataset_path) == 2:
                    author, name = dataset_path
                    base_path = f"{author}/{name}"
                else:
                    name = dataset_path[0]
                    base_path = name

                filename_base = f"{base_path}/{split}"

                if output_format == "excel":
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name='data', index=False)
                    output_files.append((f"{filename_base}.xlsx", buffer.getvalue()))

                elif output_format == "csv":
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    output_files.append((f"{filename_base}.csv", csv_data))

                elif output_format == "json":
                    json_data = df.to_json(orient='records', lines=True).encode('utf-8')
                    output_files.append((f"{filename_base}.json", json_data))

                elif output_format == "pdf":
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=10)
                    text = df.head(30).to_string()
                    for line in text.split("\n"):
                        pdf.cell(200, 5, txt=line, ln=True)
                    pdf_output = BytesIO()
                    pdf.output(pdf_output)
                    output_files.append((f"{filename_base}.pdf", pdf_output.getvalue()))

                elif output_format == "best":
                    try:
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            df.to_excel(writer, sheet_name='data', index=False)
                        output_files.append((f"{filename_base}.xlsx", buffer.getvalue()))
                        continue
                    except Exception:
                        pass

                    try:
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        output_files.append((f"{filename_base}.csv", csv_data))
                        continue
                    except Exception:
                        pass

                    try:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=10)
                        text = df.head(30).to_string()
                        for line in text.split("\n"):
                            pdf.cell(200, 5, txt=line, ln=True)
                        pdf_output = BytesIO()
                        pdf.output(pdf_output)
                        output_files.append((f"{filename_base}.pdf", pdf_output.getvalue()))
                        continue
                    except Exception:
                        pass

                    try:
                        json_data = df.to_json(orient='records', lines=True).encode('utf-8')
                        output_files.append((f"{filename_base}.json", json_data))
                        continue
                    except Exception:
                        pass

                    raw_data = dataset[split]
                    raw_json = json.dumps(raw_data[:100], indent=2).encode('utf-8')
                    output_files.append((f"{filename_base}_raw.json", raw_json))

            except Exception as e:
                st.warning(f"❌ Split '{split}' failed: {e}")

    except Exception as e:
        st.error(f"Failed to load dataset '{dataset_id}': {e}")

    return output_files

st.set_page_config(page_title="HuggingFace Extractor", layout="wide")

st.markdown("""
<div style="display: flex; flex-direction: column;">
  <div>
    <h1 style="margin-bottom: 0;">🤗 HuggingFace Dataset Extractor</h1>
  </div>
  <div style="display: flex; justify-content: flex-end;">
    <p style="margin-top: 0; font-style: italic; color: gray;">Developed by Kshitiz Sharma</p>
  </div>
</div>
""", unsafe_allow_html=True)

option = st.radio("Choose an option:", ["Single Dataset", "All Datasets by Author"])

if option == "Single Dataset":
    dataset_path = st.text_input("Enter HuggingFace dataset path (e.g., 'imdb' or 'glue/sst2')")
    format_choice = st.selectbox("Select format", ["Best (auto)", "Excel", "CSV", "PDF", "JSON"])

    if st.button("Download"):
        if dataset_path:
            with st.spinner("Downloading dataset..."):
                fmt = format_choice.lower().replace(" (auto)", "").strip()
                if fmt not in ["excel", "csv", "json", "pdf"]:
                    fmt = "best"

                files = export_dataset(dataset_path, output_format=fmt)

                if files:
                    for filename, content in files:
                        st.download_button(
                            label=f"Download {filename.split('/')[-1]}",
                            data=content,
                            file_name=filename.split("/")[-1],
                            mime="application/octet-stream"
                        )
                else:
                    st.error("No downloadable files found.")

else:
    author = st.text_input("Enter HuggingFace author name")
    if st.button("Download All Datasets"):
        if author:
            with st.spinner("Listing and downloading datasets..."):
                try:
                    author_datasets = list_datasets(author=author)
                    zip_buffer = BytesIO()

                    with zipfile.ZipFile(zip_buffer, "w") as zf:
                        for ds in author_datasets:
                            st.write(f"⏳ Processing: {ds.id}")
                            files = export_dataset(ds.id, output_format="best")
                            if files:
                                for filename, content in files:
                                    zf.writestr(filename, content)
                            else:
                                st.warning(f"⚠️ Skipped {ds.id} (no downloadable content)")

                    st.success("All available datasets downloaded successfully.")
                    st.download_button(
                        label="📦 Download ZIP File",
                        data=zip_buffer.getvalue(),
                        file_name=f"{author}_datasets.zip",
                        mime="application/zip"
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
