$(document).ready( () => {
	$(".button-collapse").sideNav();
});

$(document).ready(function() {
    $(window).scroll(function() {
        var headerHeight = $('header').height();
        if($(window).scrollTop() > headerHeight) {
            $('header.header').addClass('navbar-fixed');
        } else if($(window).scrollTop() < headerHeight) {
            $('header.header').removeClass('navbar-fixed');
        }
    });
});