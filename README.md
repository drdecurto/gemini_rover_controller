clone https://github.com/frodobots-org/earth-rovers-sdk and after setting .env start server as: 

hypercorn main:app --reload

then run code as:

python gemini_rover_controller.py --mode mission
