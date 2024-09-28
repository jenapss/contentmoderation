from concurrent import futures
import grpc
import image_classification_pb2
import image_classification_pb2_grpc
from inferencecode import make_inference

class ImageClassificationServicer(image_classification_pb2_grpc.ImageClassificationServicer):
    def ClassifyImage(self, request, context):
    # Assuming the make_inference function can handle bytes directly
        response = make_inference(image_path=request.image_path, image_file=request.image_file)
        return image_classification_pb2.ClassificationResponse(result=str(response))


    def HealthCheck(self, request, context):
        # Simulating a health check response
        return image_classification_pb2.HealthCheckResponse(status="Service is up and running")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    image_classification_pb2_grpc.add_ImageClassificationServicer_to_server(
        ImageClassificationServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
