from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer

HOST_NAME = 'localhost'
PORT = 3000

class MyHandler(BaseHTTPRequestHandler):
    def do_HEAD(s):
        s.send_response(200)
        s.send_header("Content-type", "text/html")
        s.end_headers()

    def do_POST(s):
        s.send_response(200)
        s.send_header("Content-type", "text/html")
        s.end_headers()
        file = s.rfile.read()

        s.wfile.write("<html><body><h1>" +
                            "its a cat" +
                          "</h1></body></html>")

        # predictions = getPrediction(file, torch.cuda.is_available())


        # for prediction in predictions:
        #     s.wfile.write("<html><body><h1>" +
        #                     str(prediction) +
        #                   "</h1></body></html>")

def main():
    httpd = HTTPServer((HOST_NAME, PORT), MyHandler)
    print "Server Starts - %s:%s" % (HOST_NAME, PORT)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()

if __name__ == '__main__':
    main()