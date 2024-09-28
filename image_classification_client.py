import grpc
import image_classification_pb2
import image_classification_pb2_grpc

def run():
    # Create a gRPC channel to connect to the server
    channel = grpc.insecure_channel('localhost:50051')

    # Create a stub for the ImageClassification service
    stub = image_classification_pb2_grpc.ImageClassificationStub(channel)

    # Send a request to classify an image
    image_path = "C:\\Users\\sulta\\Downloads\\snail.jpg"  # Provide the correct image path here
    with open(image_path, 'rb') as f:
        image_data = f.read()
    response = stub.ClassifyImage(image_classification_pb2.ImageRequest(image_path=image_path, image_file=image_data))

    # Access the classification result from the response
    classification_result = response.result

    # Print the classification result
    print("Classifier response:", classification_result)

    # Send a request to perform a health check
    health_response = stub.HealthCheck(image_classification_pb2.HealthCheckRequest())
    print("Health Check:", health_response.status)

if __name__ == '__main__':
    run()
