Dropzone.options.uploadWidget = {
	paramName: 'file',
	maxFiles: 1,
    dictDefaultMessage: "Drop your image file here, or click to select one",
    init: function() {
		this.on("success", function(file, resp) {
			console.log(file);
			console.log(resp);
			$('#product-tags').val(resp.tag + resp.confidence); 
		});
		this.on("addedfile", function(file, resp) {
			console.log("added");
		});
		this.on('thumbnail', function(file) {
			file.acceptDimensions();
		});
    },
    accept: function(file, done) {
		file.acceptDimensions = done;
		file.rejectDimensions = function() {
			done('The image must be at least 400 by 400 pixels in size');
		}
	}
}