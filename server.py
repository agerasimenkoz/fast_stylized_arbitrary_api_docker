#!/usr/bin/sudo python
import os
from fastapi import FastAPI, File, UploadFile, Path
from fastapi.responses import Response
import uvicorn

from image_process import image_to_byte_array
from style_transfer_model import StyleTransferModel

app = FastAPI()
model = StyleTransferModel()


@app.post('/stylized_image_style',
          # Set what the media type will be in the autogenerated OpenAPI specification.
          # fastapi.tiangolo.com/advanced/additional-responses/#additional-media-types-for-the-main-response
          responses={
              200: {
                  "content": {"image/png": {}}
              }
          },

          # Prevent FastAPI from adding "application/json" as an additional
          # response media type in the autogenerated OpenAPI specification.
          # https://github.com/tiangolo/fastapi/issues/3258
          response_class=Response
          )
def stylized_file(content_file: UploadFile = File(...), style_file: UploadFile = File(...)):
    """
    :param content_file: input image content from the post request
    :param style_file: input image style from the post request
    :return: stylized image
    """
    stylized = model.stylize_image(content_file.file.read(), style_file.file.read())

    return Response(content=image_to_byte_array(stylized),
                    media_type="image/png")


@app.post('/stylized_image_thumbnails/{style_number}',
          # Set what the media type will be in the autogenerated OpenAPI specification.
          # fastapi.tiangolo.com/advanced/additional-responses/#additional-media-types-for-the-main-response
          responses={
              200: {
                  "content": {"image/png": {}}
              }
          },

          # Prevent FastAPI from adding "application/json" as an additional
          # response media type in the autogenerated OpenAPI specification.
          # https://github.com/tiangolo/fastapi/issues/3258
          response_class=Response
          )
def stylized_file_style_thumbnails(style_number: int = Path(..., title="The ID of the item to get", ge=0, le=25)):
    """
    :param style_number: input number style prepared images [0-25] from the post request
    :return: stylized image
    """
    if os.path.isfile(f"thumbnails/style{style_number}.jpg"):
        style_path = os.path.abspath(f"thumbnails/style{style_number}.jpg")
    else:
        style_path = os.path.abspath("thumbnails/style0.jpg")
    stylized = model.stylize_image(os.path.abspath("thumbnails/chicago.jpg"),
                                   style_path)

    return Response(content=image_to_byte_array(stylized),
                    media_type="image/png")


# @app.post("/create_file/")
# async def image(image: UploadFile = File(...)):
#     print(image.file)
#     # print('../'+os.path.isdir(os.getcwd()+"images"),"*************")
#     try:
#         os.mkdir("images")
#         print(os.getcwd())
#     except Exception as e:
#         print(e)
#     file_name = os.getcwd() + "/images/" + image.filename.replace(" ", "-")
#     with open(file_name, 'wb+') as f:
#         f.write(image.file.read())
#         f.close()
#     file = jsonable_encoder({"imagePath": file_name})
#     new_image = await add_image(file)
#     return {"filename": new_image}

if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=80)
