var express  =  require( 'express' );
var multer   =  require( 'multer' );
var upload   =  multer( { dest: 'uploads/' } );
var sizeOf   =  require( 'image-size' );
var exphbs   =  require( 'express-handlebars' );
var request  =  require( 'request-promise' );
require( 'string.prototype.startswith' );

var app = express();

app.use( express.static( __dirname + '/views' ) );
app.use( express.static( __dirname + '/dropzone' ) );

app.engine( '.hbs', exphbs( { extname: '.hbs' } ) );
app.set('view engine', '.hbs');

app.get( '/', function( req, res, next ){
  return res.render( 'file-upload' );
});

app.post( '/upload', upload.single( 'file' ), function( req, res, next ) {

  if ( !req.file.mimetype.startsWith( 'image/' ) ) {
    return res.status( 422 ).json( {
      error : 'The uploaded file must be an image'
    } );
  }

  var dimensions = sizeOf( req.file.path );

  if ( ( dimensions.width < 400 ) || ( dimensions.height < 400 ) ) {
    return res.status( 422 ).json( {
      error : 'The image must be at least 400 x 400px'
    } );
  }

  const options = {
  	method: 'POST',
  	uri: 'http://localhost:3000',
  	body: req.file,
  	json: true,
  }

  request(options).then( response => {
    console.log(response)
  	// return res.status(200).send(req.file);
    //return res.status(200).send(response)
  
    var parseResponse = response.split(',');
    var sendBack = {
      'file': req.file,
      'tag': parseResponse[0], 
      'confidence': parseResponse[1]
    }


    console.log(req.file);
    return res.status(200).send(sendBack);
  })
  .catch( err => {
  	console.log(err);
  });
  
});

app.listen( 8080, function() {
  console.log( 'Express server listening on port 8080' );
});