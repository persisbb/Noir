
from django import forms


MAX_PDF_SIZE = 10 * 1024 * 1024  # 10 MB


class PDFUploadForm(forms.Form):
    document = forms.FileField(
        label="Choose a PDF",
        widget=forms.ClearableFileInput(
            attrs={"accept": ".pdf,application/pdf"}
        ),
    )

    def clean_document(self):
        uploaded_file = self.cleaned_data["document"]

        if not uploaded_file.name.lower().endswith(".pdf"):
            raise forms.ValidationError(
                "Please upload a PDF file."
            )

        if uploaded_file.size > MAX_PDF_SIZE:
            raise forms.ValidationError(
                "PDF files must be smaller than 10 MB."
            )

        # Check the actual file signature, not just its extension.
        header = uploaded_file.read(5)
        uploaded_file.seek(0)

        if header != b"%PDF-":
            raise forms.ValidationError(
                "The selected file is not a valid PDF."
            )

        return uploaded_file
