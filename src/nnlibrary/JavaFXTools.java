package nnlibrary;

import nnlibrary.hyperparameters.Functions;
import java.util.LinkedList;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import static javafx.application.Application.launch;
import javafx.event.EventHandler;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.FlowPane;
import javafx.scene.text.Text;
import javafx.stage.Screen;
import javafx.stage.Stage;
import javafx.util.Duration;

public class JavaFXTools extends Application{

    private static final LinkedList<Functions.Function<NN, Stage>> INFOLIST = new LinkedList<>();
    private static final LinkedList<NN> NNLIST = new LinkedList<>();
    private static int updateRate = 50;
    private static Timeline infoUpdater = new Timeline(new KeyFrame(Duration.millis(100), handler -> {
        if (INFOLIST.size() > 0) {
            Stage infoWindow = INFOLIST.poll().apply(NNLIST.getFirst());
            infoWindow.setTitle(NNLIST.poll().label);
            infoWindow.setX((Screen.getPrimary().getBounds().getWidth() / 5) + (Math.random() * Screen.getPrimary().getBounds().getWidth() / 5));
            infoWindow.show();
        }
    }));
    private static boolean running = false;

    /**
     *
     * @param millis Milliseconds between each update of all info panels.
     * Default is 50 milliseconds
     */
    public static void setInfoUpdateRate(int millis) {
        updateRate = millis;
    }

    private static void updaterBuilder(EventHandler handler) {
        Timeline updater = new Timeline(new KeyFrame(Duration.millis(updateRate), handler));
        updater.setCycleCount(-1);//Indefinite
        updater.play();
    }

    public static Functions.Function<NN, Stage> infoGraph(boolean mode) {
        return nn -> {
            NumberAxis xAxis = new NumberAxis();
            NumberAxis yAxis = new NumberAxis();
            xAxis.setAnimated(false);
            xAxis.setLabel("Steps");
            xAxis.setForceZeroInRange(false);
            yAxis.setAnimated(false);
            yAxis.setLabel(mode ? "Accuracy" : "Loss");
            if (mode) {
                yAxis.setForceZeroInRange(false);
            }
            XYChart.Series<Number, Number> series = new XYChart.Series<>();
            ScatterChart<Number, Number> chart = new ScatterChart<>(xAxis, yAxis);
            chart.setAnimated(false);
            chart.getData().add(series);
            updaterBuilder(handler -> {
                if (mode) {
                    series.getData().add(new XYChart.Data<>(nn.getIterations(), 1 / Math.pow(100 * Math.E, nn.getLoss())));
                } else {
                    series.getData().add(new XYChart.Data<>(nn.getIterations(), nn.getLoss()));
                }
                if (series.getData().size() > 30000) {
                    series.getData().remove(0);
                }
            });
            Stage stage = new Stage();
            stage.setScene(new Scene(chart, 600, 300));
            return stage;
        };
    }

    public static Functions.Function<NN, Stage> infoLayers = nn -> {
        ScrollPane scroll = new ScrollPane();
        FlowPane network = new FlowPane();
        scroll.setContent(network);
        int size = nn.length;
        Text[] parameters = new Text[size];
        updaterBuilder(handler -> {
            for (int i = 0; i < size; i++) {
                parameters[i] = new Text("Layer " + (i + 1) + ":\n" + nn.getLayer(i).parametersToString());
            }
            network.getChildren().clear();
            network.getChildren().addAll(parameters);
        });
        Scene scene = new Scene(scroll, 600, 300);
        Stage stage = new Stage();
        stage.setScene(scene);
        return stage;
    };

    public static void showInfo(Functions.Function<NN, Stage> info, NN nn) {
        if (!running) {
            Thread launchThread = new Thread(() -> {
                try {
                    running = true;
                    launch(JavaFXTools.class);
                } catch (IllegalStateException e) {
                    infoUpdater.setCycleCount(-1);
                    infoUpdater.play();
                    running = true;
                }
            });
            launchThread.setName("NNLib Launch Thread");
            launchThread.start();
        }
        INFOLIST.add(info);
        NNLIST.add(nn);
    }

    @Override
    public void start(Stage stage) {
        infoUpdater.setCycleCount(-1);
        infoUpdater.play();
    }
}
