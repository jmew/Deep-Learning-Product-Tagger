from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer

HOST_NAME = 'localhost'
PORT = 80

class MyHandler(BaseHTTPRequestHandler):
    def do_HEAD(s):
        s.send_response(200)
        s.send_header("Content-type", "text/html")
        s.end_headers()

    def do_POST(s):
        content = "Its a cat"
        s.send_respone(200)
        s.send_header("Content-Length", len(content))
        s.send_header("Content-Type", "text/html")
        s.end_headers()

        length = int(s.headers['Content-Length'])
        post_data = urlparse.parse_qs(s.rfile.read(length).decode('utf-8'))

        s.wfile.write(post_data)

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
