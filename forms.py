# forms.py
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField

class UploadForm(FlaskForm):
    file = FileField('Upload Image', validators=[FileRequired(), FileAllowed(['jpg', 'png', 'tif', 'tiff'], 'Images only!')])
    submit = SubmitField('Segment')
