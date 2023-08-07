from pymongo import MongoClient

# Replace <your_connection_string> with your actual MongoDB Atlas connection string
client = MongoClient("mongodb+srv://loketcom:loketdotcom@cluster.3k2noof.mongodb.net/")

# Check if the connection is successful
try:
    client.admin.command('ismaster')
    print("MongoDB Atlas connected successfully!")
except Exception as e:
    print("Failed to connect to MongoDB Atlas:", e)

# Save a random document to a collection
db = client["testdb"]  # Replace "testdb" with your desired database name
collection = db["documents"]  # Replace "documents" with your desired collection name

# Random document to save
document = {"name": "John Doe", "age": 30}

# Insert the document
result = collection.insert_one(document)

# Check if the document was inserted successfully
if result.acknowledged:
    print("Document saved successfully!")
    # Retrieve the saved document from the collection
    saved_document = collection.find_one({"_id": result.inserted_id})
    print("Saved document:", saved_document)
else:
    print("Failed to save the document.")
