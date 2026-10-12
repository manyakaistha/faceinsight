import "./globals.css";
import { ReactNode } from "react";
import { Workbench } from "../components/workbench";
export const metadata = {
  title: "FaceInsight — Vision Research Studio",
  description:
    "Explore facial age estimates, model explanations, and evaluation evidence in a local vision workbench.",
};
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Workbench>{children}</Workbench>
      </body>
    </html>
  );
}
