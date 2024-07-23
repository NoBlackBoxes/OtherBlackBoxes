# Email

Instructions to generate and send automated emails

## Create a Virtual Environment

```bash
cd <OBB/email>
mkdir _tmp
cd _tmp
python -m venv Email
cd ..
source _tmp/Email/bin/activate
```

## Install Requirements

```bash
pip install numpy pandas python-dotenv odfpy markdown
```

## Create an Environment (.env) File

```txt
LIBS_PATH="/home/kampff/NoBlackBoxes/OtherBlackBoxes/email/libs"
BASE_PATH="/home/kampff/NoBlackBoxes/OtherBlackBoxes/email"
PROTONMAIL_USERNAME='info@noblackboxes.org'
PROTONMAIL_SMTP_TOKEN='??????'
```

## Recipient (Group) List
Create an ODS file with the following format with columns with field names, should have "First Name" and must have "Email".

## Template
Create a Markdown-style template (template.md)

```txt
## Subject: This can contain named fields, like {Group ID}.
---
Dear {First Name(s)},

The email message goes here. There can be special fields, such as: {Last Name}
    
Is the following address correct?

    {Institution}, {Department}
    {Street Address}
    {Street Address Line 2}
    {City}, {State / Province}, {Postal / Zip Code}
    {Country}

You can use any named field (column) in the associated ODS file, such as {Number} or {String} fields.
You can re-use the same fields, like {Number} and {Group ID}.

You can use Markdown (MD) tags to make things **bold** or ***italic***.

Links should be clickable in plain text: https://www.noblackboxes.org

...but you can also use MD links like [this](https://www.noblackboxes.org).

Sincerely,
Adam and Elena

---
attachment_{Group ID}.jpg,attachment_{First Name}.pdf
```