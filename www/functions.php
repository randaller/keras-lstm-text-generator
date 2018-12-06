<?php

$models = array_diff( scandir( "models" ), [ ".", ".." ] );
$result = [];
foreach ( $models as $model )
{
	$model    = substr( $model, 0, -4 );
	$result[] = $model;
}
$models = $result;
asort($models);

// print_r($models);

$current = $models[0];
if ( isset( $_GET["model"] ) )
{
	$line = trim( $_GET["model"] );

	if ( in_array( $line, $models ) )
	{
		$current = $line;
	}
}

// echo( $current );

$chars_file = "chars/" . $current . ".json";
$model_file = "models/" . $current . ".bin";

$chars = file_get_contents( $chars_file );
$chars = "chars =" . trim( $chars ) . ";";

$chars = str_replace( '\n', "<br/>", $chars );

// echo $chars;

$selector = "";
foreach ( $models as $model )
{
	$selector .= "<option value=\"" . $model . "\"";

	if ( $model == $current )
	{
		$selector .= " selected";
	}

	$selector .= ">" . $model . "</option>";
}

// echo $selector;

$model_words = FALSE;
if ( stristr( $current, "_words" ) !== FALSE )
{
	$model_words = TRUE;
}

// echo $model_words;

$seq_length = 40;
if ( strpos($current, "_seq" ) !== FALSE )
{
	$rest = substr( $current, strpos($current, "_seq" )+4 );
	$rest = substr( $rest, 0, strpos($rest, "_" ) );
	$seq_length = trim($rest);
}

// echo $seq_length;
