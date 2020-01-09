package neuralnetwork;

import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.stage.Stage;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Side;
import javafx.scene.Node;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.StackedBarChart;
import javafx.scene.chart.XYChart;
import javafx.scene.chart.XYChart.Series;
import javafx.scene.paint.Color;
import javafx.util.Duration;

/**
 * An advanced line chart with a variety of actions and settable properties.
 *
 * @see javafx.scene.chart.LineChart
 * @see javafx.scene.chart.Chart
 * @see javafx.scene.chart.NumberAxis
 * @see javafx.scene.chart.XYChart
 */
public class Graph extends Application {
    LineChart<Number,Number> chart;
    private void init(Stage primaryStage) {
        Group root = new Group();
        primaryStage.setScene(new Scene(root));
        root.getChildren().add(createChart());
    }

    protected LineChart<Number, Number> createChart() {
        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        final LineChart<Number,Number> lc = new LineChart<>(xAxis,yAxis);
        // setup chart
        lc.setTitle("Basic LineChart");
        xAxis.setLabel("X Axis");
        yAxis.setLabel("Y Axis");
        // add starting data
        XYChart.Series<Number,Number> series = new XYChart.Series<>();
        series.setName("Data Series 1");
//        series.getData().add(new XYChart.Data<>(20d, 50d));
//        series.getData().add(new XYChart.Data<>(40d, 80d));
//        series.getData().add(new XYChart.Data<>(50d, 90d));
//        series.getData().add(new XYChart.Data<>(70d, 30d));
//        series.getData().add(new XYChart.Data<>(170d, 122d));
        lc.getData().add(series);
        chart = lc;
        return lc;
    }
    public LineChart<Number, Number> addData(double y){
        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        XYChart.Series<Number,Number> series = new XYChart.Series<>();
        int size = chart.getData().size();
        Number one = y;
        Number two = chart.getData().get(size).getData().get(size).getXValue();
        series.getData().add(new XYChart.Data<Number,Number>(two,one));
        
        return chart;
    }
    @Override public void start(Stage primaryStage) throws Exception {
        init(primaryStage);
        primaryStage.show();
    }
    public static void main(String[] args) {
        launch(args);
    }
}
